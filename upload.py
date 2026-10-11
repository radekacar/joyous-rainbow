"""
Upload a submission to the team's MinIO bucket, then read the bucket back and
confirm the object arrived under the exact key, at the top level, with the
right size.

    python upload.py <file>

CREDENTIALS ARE NOT IN THIS FILE AND MUST NOT BE PUT IN IT. This script is
committed; anything written into it is published. It reads credentials from:

  1. environment variables  PRC_ACCESS_KEY  and  PRC_SECRET_KEY
  2. a file  keys.txt  next to this script, two lines: access key, then secret

keys.txt is in .gitignore. Keys that have been pasted into a chat window, an
issue, or a commit are exposed and should be rotated:
https://s3-console.opensky-network.org/access-keys

Licence: GNU GPL v3.0 only.
"""
import os
import sys

TEAM_NAME = os.environ.get('PRC_TEAM', 'joyous-rainbow')
ENDPOINT = os.environ.get('PRC_ENDPOINT', 'https://s3.opensky-network.org')
BUCKET = f'prc-2026-{TEAM_NAME}'


def load_keys(here):
    ak, sk = os.environ.get('PRC_ACCESS_KEY'), os.environ.get('PRC_SECRET_KEY')
    if ak and sk:
        print('credentials: from the environment')
        return ak, sk
    kf = os.path.join(here, 'keys.txt')
    if os.path.exists(kf):
        lines = [l.strip() for l in open(kf) if l.strip()]
        if len(lines) >= 2:
            print(f'credentials: from {kf}')
            return lines[0], lines[1]
        raise SystemExit(f'{kf} must hold two non-empty lines')
    raise SystemExit('No credentials. Set PRC_ACCESS_KEY and PRC_SECRET_KEY, '
                     f'or create {kf} with the access key on line 1 and the '
                     'secret on line 2.')


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f'usage: python {os.path.basename(__file__)} <file>')
    path = sys.argv[1]
    key = os.path.basename(path)
    if not os.path.exists(path):
        raise SystemExit(f'not found: {path}')
    size = os.path.getsize(path)

    try:
        import boto3
        from botocore.client import Config
    except ImportError:
        raise SystemExit('boto3 is missing:  pip install boto3')

    here = os.path.dirname(os.path.abspath(__file__))
    ak, sk = load_keys(here)
    print(f'endpoint {ENDPOINT}\nbucket   {BUCKET}\nkey      {key}  '
          f'({size / 1e6:.1f} MB)\n')

    s3 = boto3.client('s3', endpoint_url=ENDPOINT,
                      aws_access_key_id=ak, aws_secret_access_key=sk,
                      config=Config(signature_version='s3v4'))

    with open(path, 'rb') as fh:
        s3.put_object(Bucket=BUCKET, Key=key, Body=fh)
    print('uploaded, reading the bucket back\n')

    found = None
    for o in s3.list_objects_v2(Bucket=BUCKET).get('Contents', []):
        mark = '  <--' if o['Key'] == key else ''
        print(f'  {o["Key"]:50s} {o["Size"]:>12d}{mark}')
        if o['Key'] == key:
            found = o

    print()
    if found is None:
        raise SystemExit(f'FAILED: no object named exactly {key}')
    if found['Size'] != size:
        raise SystemExit(f'FAILED: {found["Size"]} bytes uploaded, '
                         f'{size} locally')
    if '/' in found['Key'] or '%2F' in found['Key']:
        raise SystemExit(f'FAILED: the key has a path separator: {found["Key"]}')
    print(f'{key} is in {BUCKET} at the top level, {found["Size"]} bytes, '
          f'matching the local file.')


if __name__ == '__main__':
    main()
