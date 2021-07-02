import os
import json
import socket
import time

if "TF_CONFIG" in os.environ:
    print(os.environ["TF_CONFIG"])
    config = json.loads(os.environ["TF_CONFIG"])
    my_index = config['task']['index']
    ip_list = config["cluster"]["ps"]

    with open("ip_config.txt","w") as f:
        for domain_port in ip_list:
            domain, port = domain_port.split(":")
            result = None
            while result is None:
                try:
                    # connect
                    result = socket.gethostbyname(domain)
                except:
                    time.sleep(1)
            # f.write( "{} {} {}\n".format(ip, 30050, 1))
            f.write("{} {}\n".format(result, 30050))

    os.environ["MACHINE_ID"] = str(my_index)
    with open("machine_id.txt", "w") as f:
        f.write(str(my_index))

    print("Finished resolving domain")
else:
    print("Didn't find TF_CONFIG environment variable")