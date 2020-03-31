
Distributed KGE training in EC2
===============================

In this tutorial, we will introduce how to perform DGL-KE distributed training on AWS EC2
and demonstrate how to setup two EC2 instances to train our knowledge graph embeddings on **\ *FB15k*\ **.
Note that, the FB15k is just a small dataset as our toy demo. You can try much bigger datasets
such as **\ *Freebase*\ ** using the similar instructions.

**Prepare your machines**
[Image: image.png]


#. Click **\ *Launch Instance *\ **\ button to launch new instance on AWS EC2.

[Image: image.png]


#. Input **\ *Deep learning base AMI *\ **\ in search bar\ **\ *, *\ **\ and\ **\ * *\ **\ select the first one **\ *Deep learning Base AMI (Ubuntu 18.04).*\ **

[Image: image.png]In this demo, we choose **\ *r5dn.24xlarge *\ **\ instance as our testbed. This instance has 48 cores and 768GB memory, which can be used to train very large embeddings like Freebase. Also, this instance has powerful network hardware with 100 Gigabit bandwidth. High bandwidth can significantly reduce the distributed training overhead.


#. Click **\ *Next: Configure instance detail *\ **\ button at the bottom of this page and you will see the following page.
   [Image: image.png]Set **\ *Number of instances*\ ** to **\ *2*\ **\ ,  and **\ *Add instance to placement group. *\ **\ Choose\ **\ * placement group strategy *\ **\ as\ **\ * cluster *\ **\ and\ **\ * *\ **\ set a new name (\ **\ *my_cluster*\ **\ ) for our cluster. This step will place our instances into the same cluster on EC2 datacenter and achieve the highest network throughput.

User can follow the rest of the setup steps by default options.

**Install packages**

After launching the instances, users can install necessary packages on these two machines.


#. Install ``pytorch`` by the following command:

``sudo pip3 install torch``


#. Install ``dglke`` by the following command:

``sudo pip3 install dglke``

**Prepare dataset**


#. Create a new directory called ``my_task``  on machine_0.

``mkdir my_task``


#. Download ``FB15k`` dataset and partition it into **\ *2*\ ** parts.

``cd my_task``
``dglke_partition --dataset FB15k -k 2 --data_path ~/my_task``

Note that, in this task we have only two machines, so we set ``-k`` argument to ``2``\ , which is equal to your machine number. You can change the ``—dataset`` to ``Freebase`` if your want to test a much bigger dataset.


#. Create a new file called\ ``ip_config.txt``  in ``my_task`` folder and write the private IP of your instance into it.

[Image: image.png]You can find the Private IP of your instance in the **\ *Description*\ ** page. For example, the ``ip_config.txt`` of mine is as follow:

``172.31.24.245 30050 8``
``172.31.22.41 30050 8``
(DO NOT leave empty lines in this file)

In this\ ``ip_config.txt``\ , we have two lines and each line represent one machine. ``30050`` is the port and ``8`` is the number of kvstore server processes on each machine. Because our instance has 48 cores, we set the number of kvstore server processes to 8. And the rest of cores will be used by client (trainer).


#. scp the ``my_task`` folder to another machine:

``scp -i your_ssh_key -r ~/my_task 172.31.22.41:~``


#. Run the following command on ``machine_0`` to start a distributed task:

.. code-block::

   dglke_dist_train --path ~/my_task --ssh_key ~/mctt.pem --ip_config ~/my_task/ip_config.txt \
   --num_client_proc 16 --model TransE_l2 --dataset FB15k --data_path ~/my_task --hidden_dim 400 \
   --gamma 19.9 --lr 0.25 --batch_size 1000 --neg_sample_size 200 --max_step 1000 --log_interval 100 \
   --batch_size_eval 16 --test -adv --regularization_coef 1.00E-07 --no_save_emb --num_thread 1

Note that, all the path in our arguments are **\ *absolute path. *\ **\ Also, you can remove the\ **\ * *\ **\ ``—ssh_key``\ **\ * *\ **\ argument if your machines can ``ssh`` to each other directly.

If this task executed successfully, you will see the following message:

``...``
``Pull model from kvstore: 99 / 100 ...``
``Pull model from kvstore: 100 / 100 ...``
``Total train time 63.061 seconds``
``Run test, test processes: 16``
``-------------- Test result --------------``
``Test average MRR : 0.6200464812800418``
``Test average MR : 40.713133348004945``
``Test average HITS@1 : 0.4704508134279088``
``Test average HITS@3 : 0.7423270301840158``
``Test average HITS@10 : 0.8478779773492915``

``Exit KVStore service 4, solved message count: 10084``
``Exit KVStore service 2, solved message count: 10025``
``...``

----
