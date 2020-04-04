# -*- coding: utf-8 -*-
"""
Spyder Editor
@author: Rajiv Sambasivan
"""
import logging

from requests.adapters import HTTPAdapter
from requests import Session

from arango.response import Response
from arango.http import HTTPClient
import os
from os.path import dirname, join


class CustomHTTPClient(HTTPClient):
    """My custom HTTP client with cool features."""
    def __init__(self, username, password):
        # Initialize your logger.
        self._logger = logging.getLogger('my_logger')
        self.username = username
        self.password = password
        # self.cert_name = 'ca-b9b556df.crt'
        self.path_to_cert = os.path.join(os.path.dirname(__file__), "cert/")
        self.cert = os.path.join(self.path_to_cert,
                                 os.listdir(self.path_to_cert)[0])

    def create_session(self, host):
        session = Session()

        # Add request header.
        session.headers.update({'x-my-header': 'true'})

        session.auth = (self.username, self.password)

        # Enable retries.
        adapter = HTTPAdapter(max_retries=5)
        session.mount('https://', adapter)

        return session

    def send_request(self,
                     session,
                     method,
                     url,
                     params=None,
                     data=None,
                     headers=None,
                     auth=None):
        # Add your own debug statement.
        self._logger.debug('Sending request to {}'.format(url))

        # Send a request.
        response = session.request(method=method,
                                   url=url,
                                   params=params,
                                   data=data,
                                   headers=headers,
                                   verify=self.cert)
        self._logger.debug('Got {}'.format(response.status_code))

        # Return an instance of arango.response.Response.
        return Response(
            method=response.request.method,
            url=response.url,
            headers=response.headers,
            status_code=response.status_code,
            status_text=response.reason,
            raw_body=response.text,
        )
