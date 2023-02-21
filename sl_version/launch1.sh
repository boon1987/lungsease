#! /bin/bash


../../scripts/bin/run-sn -d --rm --name=sn1 --network=host-1-net --host-ip=10.60.3.51 --sentinel --sn-p2p-port=30303 --sn-api-port=30304 --key=lung-pyt-cert/cert/sn-1-key.pem --cert=lung-pyt-cert/cert/sn-1-cert.pem --capath=lung-pyt-cert/cert/ca/capath --apls-ip=10.60.3.50 -e SL_DEVMODE_KEY=REVWTU9ERS0yMDIzLTAyLTA44baf51a9d425ab9f6b2440b563b7c677a576a87e465464f01a14a3b724dbaa3a &&

../../scripts/bin/run-swop -d --rm --name=swop1 --network=host-1-net --usr-dir=swop --profile-file-name=swop_profile1.yaml --key=lung-pyt-cert/cert/swop-1-key.pem --cert=lung-pyt-cert/cert/swop-1-cert.pem --capath=lung-pyt-cert/cert/ca/capath -e SWOP_KEEP_CONTAINERS=True -e http_proxy= -e https_proxy= --apls-ip=10.60.3.50 &&

../../scripts/bin/run-swci -ti --rm --name=swci1 --network=host-1-net --usr-dir=swci --init-script-name=swci-init --key=lung-pyt-cert/cert/swci-1-key.pem --cert=lung-pyt-cert/cert/swci-1-cert.pem --capath=lung-pyt-cert/cert/ca/capath -e http_proxy= -e https_proxy= --apls-ip=10.60.3.50 -e SWCI_TASK_MAX_WAIT_TIME=2880
