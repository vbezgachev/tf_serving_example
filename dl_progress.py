from tqdm import tqdm


class DLProgress(tqdm):
    '''
    Helper class to show the download progress in the terminal
    '''

    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
