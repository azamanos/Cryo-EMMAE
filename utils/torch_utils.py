import torch

def save_checkpoint(state, epoch, filename='check.pth.tar'):
    '''
    Function to save checkpoint of a pytorch model.

    Parameters
    ----------
    state : torch.model
        state of the model after the end of an epoch.

    epoch : int
        number of epoch that corresponds to the models state.

    filename : str
        filename to save the checkpoint, default 'check.pth.tar'.

    Returns
    -------
    '''
    print(f'=> Saving checkpoint, epoch {epoch}.')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, epoch, print_load = True):
    '''
    Function to load checkpoint of a pytorch model.

    Parameters
    ----------
    checkpoint : torch.load
        loaded checkpoint of the model.

    model : torch.model
        pytorch model

    epoch : int
        number of epoch that corresponds to the models state.

    Returns
    -------
    model : torch.model
        pytorch model with loaded weights of the checkpoint.
    '''
    if print_load:
        print(f"=> Loading checkpoint, epoch {epoch}.")
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        model.load_state_dict(checkpoint)
