import sys
import os
from dotenv import load_dotenv


project_dir = os.path.expanduser('~/apps')  # adjust as appropriate
load_dotenv(os.path.join(project_dir, '.env'))

path = '/home/jungsnn1029/apps'
if path not in sys.path:
    sys.path.append(path)


from apps import app as application
# if on local computer, I'd do application.run()
