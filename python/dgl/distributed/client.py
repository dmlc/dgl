"""Functions used by client."""

import dgl

def connect_to_server(server_namebook):
    """Connect this client to server.

    Parameters
    ----------
    server_namebook : ...
        Server names.

    Raises
    ------
    ConnectionError : If anything wrong with the connection.
    """
    pass

def finalize():
    """Release resources of this client."""
    pass

def get_server_namebook():
    """Get the servers this client connects to.

    Returns
    -------
    server_namebook : ...
        Server names.
    """
    pass

def shutdown_servers():
    """Issue commands to remote servers to shut them down.

    Raises
    ------
    ConnectionError : If anything wrong with the connection.
    """
    pass
