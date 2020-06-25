import torch
import os


def save_model(args, exp_dir, model, optimizer, best_loss,folder_name):
    torch.save(
        {
            'epoch': args.num_epochs,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_loss': best_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )

    torch.save(model.state_dict(), os.path.join(args.exp_dir, folder_name, 'Model.pt'))
    # if is_new_best:
    #     shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')