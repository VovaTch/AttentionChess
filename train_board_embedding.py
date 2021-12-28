import argparse

import torch

from model.attchess import BoardEmbTrainNet
from data_loaders.dataloader import BoardEmbeddingLoader


def main(args):

    # Model
    model = BoardEmbTrainNet()
    model = model.train().to(args.device)

    # Data loader
    data_loader = BoardEmbeddingLoader(batch_size=6)
    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True, lr=args.lr, weight_decay=args.wd)

    # losses
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(args.epochs):
        print(f'EPOCH {epoch + 1} ----------------------------------====>>>')
        for idx, (board_emb, tgt_piece_prob, tgt_flags) in enumerate(data_loader):

            board_emb, tgt_piece_prob, tgt_flags = board_emb.to(args.device), \
                                                   tgt_piece_prob.to(args.device), \
                                                   tgt_flags.to(args.device)
            pred_piece_prob, pred_flags = model(board_emb)
            optimizer.zero_grad()

            loss = loss_ce(pred_piece_prob, tgt_piece_prob) + loss_bce(pred_flags, tgt_flags.float())
            loss.backward()
            optimizer.step()

            print(f'[STEP {idx}/{len(data_loader)}] Loss: {loss:.4f}')

    torch.save(model.backbone_embedding.state_dict(), 'model/board_embedding.pth')
    print('Saved model.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of board embedding script.')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device of the net.')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay.')
    arg = parser.parse_args()
    main(arg)
