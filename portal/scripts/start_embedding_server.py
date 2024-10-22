import atexit
import os
import pickle
import sys
from subprocess import Popen
from typing import Union

import zmq

from portal.model.torch_modules import embedding_model_to_dimension_and_pooling


class EmbeddingServerStarter:
    def __init__(self):
        self.sentence_embedding_model_name: Union[None, str] = None
        self.zmq_port: Union[None, int] = None
        self.process: Union[None, Popen] = None

    def start(self, sentence_embedding_model_name: str, zmq_port=5555):
        self.sentence_embedding_model_name = sentence_embedding_model_name
        self.zmq_port = zmq_port
        self.restart()

    def restart(self):
        assert self.sentence_embedding_model_name is not None, 'The starter service was never initialized!'
        print('Start async server to compute embedding')
        # Use sys.executable to make sure we use the same python interpreter as the current process.
        cwd = None
        if not os.path.exists('portal/scripts/zmq_server.py'):
            cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

            if not os.path.exists(cwd + '/portal/scripts/zmq_server.py'):
                raise FileNotFoundError('Could not find portal/scripts/zmq_server.py.')

        commands = [sys.executable, '-m', 'portal.scripts.zmq_server', '--port', str(self.zmq_port)]
        self.process = Popen(commands, cwd=cwd)

        # Wait until the server is ready - send a test message without timeout
        print('\n\nWaiting for server to start...')
        socket = zmq.Context().socket(zmq.REQ)
        socket.connect(f'tcp://localhost:{self.zmq_port}')
        socket.send(pickle.dumps(['hello world']))
        socket.recv()
        socket.close()
        del socket
        print('Done!')

        # Register a function that kills the subprocess when the main process exits.
        # This is needed because otherwise the subprocess would keep running even
        # after the main process exits.
        def kill_subprocess():
            self.process.terminate()
            print('Subprocess terminated')

        atexit.register(kill_subprocess)

    def test(self):
        assert self.sentence_embedding_model_name is not None and self.zmq_port is not None, 'The starter service was never initialized!'
        print(f'Running test embedding job on port {self.zmq_port}...')
        socket = zmq.Context().socket(zmq.REQ)
        socket.connect(f'tcp://localhost:{self.zmq_port}')
        # Timeout after 10 seconds
        socket.setsockopt(zmq.RCVTIMEO, 10000)
        socket.setsockopt(zmq.LINGER, 0)

        serialized_data = pickle.dumps(['hello', 'world'])
        socket.send(serialized_data)

        try:
            response = socket.recv()
        except zmq.error.Again as e:
            raise ValueError('No response from server, it did not start correctly.') from e

        expected_size = 4 * embedding_model_to_dimension_and_pooling[self.sentence_embedding_model_name][0]
        if len(response) != expected_size:
            raise ValueError(f'Expected {expected_size} bytes of response, got {len(response)} bytes.')

    def terminate(self):
        self.process.terminate()


embedding_server_starter = EmbeddingServerStarter()

if __name__ == '__main__':
    embedding_server_starter.start('sentence-transformers/all-MiniLM-L6-v2', ['wikipedia'])
    embedding_server_starter.test()
