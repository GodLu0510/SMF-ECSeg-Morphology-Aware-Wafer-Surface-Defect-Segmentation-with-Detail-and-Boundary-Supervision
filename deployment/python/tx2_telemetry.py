"""TX2 power rails/GPU readings and raw tegrastats, extending common telemetry."""
import glob
import os
from pathlib import Path
import shutil
import subprocess
import ecseg_telemetry as base
from telemetry import read_text
from ecseg_common import dump_json

class Monitor(base.ExtendedMonitor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for path in ('/sys/devices/17000000.gp10b/load','/sys/class/devfreq/17000000.gp10b/load'):
            if read_text(path) is not None:
                self.add_sensor('gpu_load_percent',path,.1)
                break
        for path in sorted(glob.glob('/sys/class/devfreq/*gp10b*/cur_freq')):
            self.add_sensor('gpu_frequency_mhz',path,1e-6)
        # TX2 ina3221x exports power in milliwatts, unlike hwmon microwatts.
        for path in sorted(glob.glob('/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power*_input')):
            p=Path(path);index=p.name[len('in_power'):].split('_')[0]
            label=(read_text(p.parent/('rail_name_'+index)) or ('channel_'+index)).strip().replace(' ','_')
            key='power_ina3221_'+p.parent.parent.name.replace('-','_')+'_'+label+'_w'
            self.add_sensor(key,path,.001)
            self.sources[key]['scope']='Direct TX2 rail sensor; rail labels retained; do not sum overlapping rails'

class Sampler(base.Sampler):
    def __init__(self,out,interval,npu_path=None,power_path=None,power_scale=1):
        super().__init__(out,interval,npu_path,power_path,power_scale)
        self.monitor=Monitor(os.getpid(),npu_path,power_path,power_scale)
        self.tegra=None;self.tegra_log=None
    def start(self):
        executable=shutil.which('tegrastats') or '/usr/bin/tegrastats'
        if Path(executable).is_file():
            self.tegra_log=(self.out/'tegrastats.log').open('w',encoding='utf-8')
            try:
                self.tegra=subprocess.Popen([executable,'--interval',str(max(100,int(self.interval*1000)))],
                    stdout=self.tegra_log,stderr=subprocess.STDOUT,universal_newlines=True)
            except OSError as exc:
                self.errors.append({'scope':'tegrastats','error':repr(exc)})
        super().start()
    def close(self):
        try:super().close()
        finally:
            if self.tegra is not None:
                self.tegra.terminate()
                try:self.tegra.wait(timeout=3)
                except subprocess.TimeoutExpired:self.tegra.kill();self.tegra.wait()
            if self.tegra_log is not None:self.tegra_log.close()
    def summary(self,start,end):
        result=super().summary(start,end)
        result['scope']='TX2 device-wide CPU/GPU/temperature/power rails; RSS includes Python/TensorRT; CUDA unified memory accounting may overlap'
        return result

def board_snapshot(out,suffix):
    base.board_snapshot(out,suffix)
    results={}
    for command in (['/usr/sbin/nvpmodel','-q','--verbose'],['/usr/bin/jetson_clocks','--show']):
        try:
            r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=10)
            results[' '.join(command)]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        except (OSError,subprocess.TimeoutExpired) as exc:results[' '.join(command)]={'error':repr(exc)}
    results['L4T']=read_text('/etc/nv_tegra_release')
    dump_json(Path(out)/('jetson_'+suffix+'.json'),results)
