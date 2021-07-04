import os
import json
import socket
import time


def generate_ip_config():
    if "TF_CONFIG" in os.environ:
        print(os.environ["TF_CONFIG"])
        config = json.loads(os.environ["TF_CONFIG"])
        my_index = config['task']['index']
        ip_list = config["cluster"]["ps"]

        with open("ip_config.txt", "w") as f:
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
        raise Exception("Didn't find TF_CONFIG environment variable")
    return {"num_machines": len(ip_list), "machine_id": my_index}


def generate_script(machine_id, num_servers_per_machine, num_machines):
    total_server_count = num_servers_per_machine * num_machines
    num_clients = num_machines
    server_command_list = []
    client_command_list = []
    for i in range(num_servers_per_machine):
        server_id = i+machine_id*num_servers_per_machine
        server_command_list.append(f"python dist_sample_server.py --server-id {server_id} --num-servers {num_servers_per_machine} --num-clients {num_clients}")
    # Current assume only 1 client per machine
    client_command_list.append(f"python dist_sample_client.py --machine-id {machine_id}")
    full_commands = " & ".join(server_command_list+client_command_list)
    print("==========================Full command==========================")
    print(full_commands)
    print("==========================Full command==========================")
    with open("run.sh", "w") as f:
        f.write(full_commands)

stats_dict = generate_ip_config()
generate_script(stats_dict["machine_id"], 1, stats_dict["num_machines"])