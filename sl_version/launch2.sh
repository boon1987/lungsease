#! /bin/bash

../../scripts/bin/run-sn -d --rm --name=sn2 --network=host-2-net --host-ip=10.60.3.52 --sentinel-ip=10.60.3.51 --sn-p2p-port=30303 --sn-api-port=30304 --key=lung-pyt-cert/cert/sn-2-key.pem --cert=lung-pyt-cert/cert/sn-2-cert.pem --capath=lung-pyt-cert/cert/ca/capath --apls-ip=10.60.3.50 -e SL_DEVMODE_KEY=REVWTU9ERS0yMDIzLTAyLTA44baf51a9d425ab9f6b2440b563b7c677a576a87e465464f01a14a3b724dbaa3a

../../scripts/bin/run-swop -d --rm --name=swop2 --network=host-2-net --usr-dir=swop --profile-file-name=swop_profile2.yaml --key=lung-pyt-cert/cert/swop-2-key.pem --cert=lung-pyt-cert/cert/swop-2-cert.pem --capath=lung-pyt-cert/cert/ca/capath -e SWOP_KEEP_CONTAINERS=True -e http_proxy= -e https_proxy= --apls-ip=10.60.3.50
