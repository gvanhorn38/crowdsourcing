import sys

# https://gist.github.com/vladignatyev/06860ec2040cb497f0f3
def progress_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percentage_complete = round(100.0 * count / float(total), ndigits=1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percentage_complete, '%', status))
    sys.stdout.flush()