import collections

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
def flatten_map(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == '__main__':
    a= {'name': 'loop-px8xyw', 'jobStatus': {'username': '_4a554b6acb4f4a558d8e1ff0df394952', 'state': 'STOPPED', 'subState': 'FRAMEWORK_COMPLETED', 'executionType': 'STOP', 'retries': 0, 'createdTime': 1547436601064, 'completedTime': 1547437680353, 'appId': 'application_1546606438250_8466', 'appProgress': 1, 'appTrackingUrl': 'http://10.11.3.2:8088/proxy/application_1546606438250_8466/', 'appLaunchedTime': 1547436659732, 'appCompletedTime': 1547437680353, 'appExitCode': 214, 'appExitDiagnostics': 'UserApplication killed due to StopFrameworkRequest', 'appExitType': 'NON_TRANSIENT', 'virtualCluster': 'default'}, 'taskRoles': {'setup': {'taskRoleStatus': {'name': 'setup'}, 'taskStatuses': [{'taskIndex': 0, 'containerId': None, 'containerIp': None, 'containerPorts': {}, 'containerGpus': None, 'containerLog': None}]}}}
    print(flatten(a))
    b= {'jobName': 'loop-px8xyw', 'image': '10.11.3.8:5000/user-images/cy-pytorch0.4.0-py36:cyclegan', 'gpuType': 'debug', 'retryCount': 0, 'taskRoles': [{'name': 'setup', 'memoryMB': 8000, 'shmMB': 64, 'taskNumber': 1, 'cpuNumber': 4, 'gpuNumber': 1, 'minFailedTaskCount': None, 'minSucceededTaskCount': None, 'command': 'while true; do sleep 60m;  done'}]}
    print(flatten(b))