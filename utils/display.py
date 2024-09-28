def display_progress_bar(progress, total_size, done=False):
    """Function responsible for displaying the progress bar.
    
    If done is True, shows the completed message instead of the progress.
    """
    if done:
        print(f"\r[{'=' * 50}] {total_size / (1024*1024):.2f} MB - Done!", end='\n')
    else:
        done_percentage = int(50 * progress / total_size)
        print(f"\r[{'=' * done_percentage}{' ' * (50 - done_percentage)}] {progress / (1024*1024):.2f}/{total_size / (1024*1024):.2f} MB", end='')