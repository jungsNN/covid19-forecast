import sys
import os
from dotenv import load_dotenv

# /var/www/wsgi.py configuration
project_dir = os.path.expanduser('~/mysite')  # adjust as appropriate
load_dotenv(os.path.join(project_dir, '.env'))

path = '/home/jungsnn1029/mysite'
if path not in sys.path:
    sys.path.append(path)


from apps import app as application  # noqa
# if on local computer, I'd do application.run()
