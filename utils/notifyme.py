import socket
from urllib import request, parse



__key='wf96HV'


class BadRequest(Exception):
    """Raised when API thinks that title or message are too long."""
    pass


class UnknownError(Exception):
    """Raised for invalid responses."""
    pass

def _handle_response(response):
    """Raise error if message was not successfully sent."""
    if response.json()['status'] == 'BadRequest' and response.json()['message'] == 'Title or message too long':
        raise BadRequest

    if response.json()['status'] != 'OK':
        raise UnknownError

    response.raise_for_status()



def send_notification(msg, title=None, key=None, event=None):

    if key is None: key = __key

    if title is None:
        title = socket.gethostname()


    data = parse.urlencode({'key': key, 'title': title, 'msg': msg, 'event': event}).encode()
    req = request.Request("https://api.simplepush.io/send", data=data)
    request.urlopen(req)



