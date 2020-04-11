
import code
import signal
import traceback

# From https://stackoverflow.com/questions/132058/showing-the-stack-trace-from-a-running-python-application
#
# Call listen() when starting up the application
# When the application hangs or you want to interrupt
# and debug it, send the SIGUSR1 signal:
#
# kill -SIGUSR1 <pid>

def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d={'_frame':frame}         # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message  = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)

def listen( s=signal.SIGUSR1):
    signal.signal(s, debug)  # Register handle

 

