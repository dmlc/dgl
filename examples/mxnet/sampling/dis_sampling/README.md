### Demo for Distributed Sampler

First we need to change the `--ip` and `--port` in `run_trainer.sh` and `run_sampler.sh`

Then we need to start trainer node:

```
./run_trainer.sh
```

When you see the message:

```
[04:48:20] /home/ubuntu/dgl_da/src/graph/network/socket_communicator.cc:68: Bind to 127.0.0.1:2049
[04:48:20] /home/ubuntu/dgl_da/src/graph/network/socket_communicator.cc:74: Listen on 127.0.0.1:2049, wait sender connect ...
```

you can start sampler:

```
./run_sampler.sh
```