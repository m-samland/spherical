#!/usr/bin/env python
from __future__ import print_function

# from IPython.config.loader import Config
import inspect

# First import the embed function
from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config.loader import Config

# Now create the IPython shell instance. Put ipshell() anywhere in your code
# where you want it to open.


try:
    get_ipython
except NameError:
    nested = 0
    cfg = Config()
    prompt_config = cfg.PromptManager
    prompt_config.in_template = 'In <\\#>: '
    prompt_config.in2_template = '   .\\D.: '
    prompt_config.out_template = 'Out<\\#>: '
else:
    # print("Running nested copies of IPython.")
    # print("The prompts for the nested copy have been modified")
    cfg = Config()
    nested = 1

# Messages displayed when I drop into and exit the shell.
banner_msg = ("\n**Nested Interpreter:\n"
              "Hit Ctrl-D to exit interpreter and continue program.\n"
              "Note that if you use %kill_embedded, you can fully deactivate\n"
              "This embedded instance so it will never turn on again")
exit_msg = '**Leaving Nested interpreter'

# Wrap it in a function that gives me more context:


def ipsh():
    ipshell = InteractiveShellEmbed(config=cfg, banner1=banner_msg, exit_msg=exit_msg)

    frame = inspect.currentframe().f_back
    msg = 'Stopped at {0.f_code.co_filename} at line {0.f_lineno}'.format(frame)

    # Go back one level!
    # This is needed because the call to ipshell is inside the function ipsh()
    ipshell(msg, stack_depth=2)
