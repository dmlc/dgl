#!/bin/bash

# generate ip_config.txt
NODE_TYPE="child"
if [ "${AWS_BATCH_JOB_MAIN_NODE_INDEX}" == "${AWS_BATCH_JOB_NODE_INDEX}" ]; then
    NODE_TYPE="main"
fi
echo "Running on ${NODE_TYPE} node..."

# wait for all nodes to report IP
wait_for_nodes () {
    echo "Main node is waiting for all nodes to report IP..."
    touch ${IP_CONFIG}
    echo $(hostname -i) >> ${IP_CONFIG}
    IP=$(hostname -i)
    echo "Main node's IP [${IP}] has been recorded."
    NUM_LINES=$(sort ${IP_CONFIG} | uniq | wc -l)
    while [ "${AWS_BATCH_JOB_NUM_NODES}" -gt "${NUM_LINES}" ]
    do
        echo "${NUM_LINES} out of ${AWS_BATCH_JOB_NUM_NODES} nodes joined, check again in 5 second."
        sleep 5
        NUM_LINES=$(sort ${IP_CONFIG} | uniq | wc -l)
    done
    echo "All nodes successfully joined: $(cat ${IP_CONFIG})"
}

# report IP to main node
report_to_main () {
    IP=$(hostname -i)
    CMD="cat ${IP_CONFIG} && echo ${IP} >> ${IP_CONFIG}"
    ssh -o StrictHostKeyChecking=no -p ${SSH_PORT} ${AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS} ${CMD}
    echo "Node~${AWS_BATCH_JOB_NODE_INDEX} has reported ${IP} to main node."
}

# share IP list with all nodes
share_to_nodes () {
    for target in $(cat ${IP_CONFIG})
    do
        scp -o StrictHostKeyChecking=no -P ${SSH_PORT} ${IP_CONFIG} ${target}:${IP_CONFIG}
    done
    echo "Shared IP to all nodes."
}

# non-main nodes waits for IPs share
wait_for_ip_share () {
    while [ ! -f "${IP_CONFIG}" ]
    do
        echo "Node~${AWS_BATCH_JOB_NODE_INDEX} is waiting for IP share from main node. Sleep 5 second."
        sleep 5
    done
}

echo "Current node is ${NODE_TYPE}"

case ${NODE_TYPE} in
    "main")
        wait_for_nodes
        share_to_nodes
        ;;
    "child")
        report_to_main
        wait_for_ip_share
        ;;
esac

echo "-----IP list on Node~${AWS_BATCH_JOB_NODE_INDEX}: $(cat ${IP_CONFIG})."

exit 0
