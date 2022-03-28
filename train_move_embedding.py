import argparse

import torch

from model.attchess import MoveEmbTrainNet
from data_loaders.dataloader import MoveEmbeddingLoader


def main(args):  # TODO: Convert to training move embeddings

    # Model
    model = MoveEmbTrainNet(emb_size=args.em_size)
    model = model.train().to(args.device)

    # Data loader
    data_loader = MoveEmbeddingLoader(batch_size=1024)
    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True, lr=args.lr, weight_decay=args.wd)

    # losses
    loss_mse = torch.nn.MSELoss()
    loss_ce = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        print(f'EPOCH {epoch + 1} ----------------------------------====>>>')
        for idx, (move_emb, tgt_coor, tgt_prom) in enumerate(data_loader):

            move_emb, tgt_coor, tgt_prom = move_emb.to(args.device), \
                                                   tgt_coor.to(args.device), \
                                                   tgt_prom.to(args.device)
            pred_coord, pred_prom = model(move_emb)
            optimizer.zero_grad()

            loss = loss_mse(pred_coord, tgt_coor) + loss_ce(pred_prom, tgt_prom)
            loss.backward()
            optimizer.step()

            print(f'[STEP {idx}/{len(data_loader)}] Loss: {loss:.4f}')

    torch.save(model.query_embedding.state_dict(), 'model/move_embedding.pth')
    print('Saved model.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of board embedding script.')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device of the net.')
    parser.add_argument('-e', '--epochs', type=int, default=1000, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-3, help='Weight decay.')
    parser.add_argument('--em_size', type=int, default=32, help='Size of the embedding layer')
    arg = parser.parse_args()
    main(arg)
