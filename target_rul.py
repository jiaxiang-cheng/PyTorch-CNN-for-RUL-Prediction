def target_rul(max_cycle, cycle, func):

    if func == "linear":
        target = min(max_cycle - cycle, 130)
    else:
        target = 100

    return target
