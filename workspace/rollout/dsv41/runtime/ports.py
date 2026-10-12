"""Separate HCCL host and device listeners for resident trainer/rollout processes."""
import os
import re

DEFAULTS={
    'TRAIN_HCCL_NPU_PORT_RANGE':'20000-20999',
    'ROLLOUT_HCCL_NPU_PORT_RANGE':'21000-21999',
    'TRAIN_HCCL_HOST_PORT_RANGE':'42000-42999',
    'ROLLOUT_HCCL_HOST_PORT_RANGE':'43000-43999',
}


def hccl_environment(environment,role):
    if role not in ('train','rollout'):raise ValueError('invalid HCCL role')
    env=dict(environment);ranges={}
    for key,default in DEFAULTS.items():
        value=env.get(key,default)
        match=re.fullmatch(r'(\d+)-(\d+)',value)
        if not match:raise ValueError(key+' requires start-end')
        lo,hi=map(int,match.groups())
        if not 1024<=lo<=hi<=65535:raise ValueError('invalid port range '+key)
        ranges[key]=(lo,hi);env[key]=value
    for kind in ('NPU','HOST'):
        a=ranges[f'TRAIN_HCCL_{kind}_PORT_RANGE'];b=ranges[f'ROLLOUT_HCCL_{kind}_PORT_RANGE']
        if max(a[0],b[0])<=min(a[1],b[1]):raise ValueError('train/rollout HCCL '+kind+' ports overlap')
        env[f'HCCL_{kind}_SOCKET_PORT_RANGE']=env[f'{role.upper()}_HCCL_{kind}_PORT_RANGE']
    return env


def configure(role):
    selected=hccl_environment(os.environ,role)
    for key in (*DEFAULTS,'HCCL_NPU_SOCKET_PORT_RANGE','HCCL_HOST_SOCKET_PORT_RANGE'):
        os.environ[key]=selected[key]
    return {key:selected[key] for key in ('HCCL_NPU_SOCKET_PORT_RANGE','HCCL_HOST_SOCKET_PORT_RANGE')}
