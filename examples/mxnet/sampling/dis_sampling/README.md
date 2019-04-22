### Demo for Distributed Sampler

First we need to change the `--ip` in `run_trainer.sh` and `run_sampler.sh` for your own environemnt.

Then we need to start trainer node:

```
./run_trainer.sh
```

When you see the message:

```
[04:48:20] .../socket_communicator.cc:68: Bind to 127.0.0.1:2049
[04:48:20] .../socket_communicator.cc:74: Listen on 127.0.0.1:2049, wait sender connect ...
```

then, you can start sampler:

```
./run_sampler.sh
``` 