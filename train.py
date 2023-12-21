import torch

def train(train_loader, val_loader, generator, discriminator, criterion, g_opt, d_opt, device, task, num_epochs=50):
    with open(f'log_{task}.txt','w') as f:
        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0

            #train
            generator.train()
            discriminator.train()
            for num_batch, (images, targets, texts_features) in enumerate(train_loader):
                images = images.to(device)
                targets = targets.to(device)
                texts_features = texts_features.to(device)

                real_labels = torch.ones(images.size(0), 1).to(device)
                fake_targets = generator(images, texts_features)
                fake_labels = torch.zeros(images.shape[0], 1).to(device) 
                outputs = discriminator(targets)
                r_loss = criterion(outputs, real_labels)

                #training d
                d_opt.zero_grad()
                outputs = discriminator(fake_targets.detach())
                f_loss = criterion(outputs, fake_labels)
                d_loss = r_loss + f_loss
                d_loss.backward()
                d_opt.step()

                #training g
                g_opt.zero_grad()
                outputs = discriminator(fake_targets)
                g_loss = criterion(outputs, real_labels)
                g_loss.backward()
                g_opt.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

                # if (num_batch + 1) % 10 == 0:
                #     # print(f'epoch [{epoch + 1}/{num_epochs}], step [{num_batch + 1}/{len(train_loader)}], '
                #     #     f'g_loss: {epoch_g_loss / (num_batch + 1)}, d_loss: {epoch_d_loss / (num_batch + 1)}')
                #     f.write(f'epoch [{epoch + 1}/{num_epochs}], step [{num_batch + 1}/{len(train_loader)}], g_loss: {epoch_g_loss / (num_batch + 1)}, d_loss: {epoch_d_loss / (num_batch + 1)}\n')
            # validate
            generator.eval()
            discriminator.eval()
            with torch.no_grad():
                val_loss = 0.0
                for images, targets, texts_features in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)
                    texts_features = texts_features.to(device)
                    fake_targets = generator(images, texts_features)
                    outputs = discriminator(fake_targets)
                    val_loss += criterion(outputs, torch.ones(images.size(0), 1).to(device)).item()
            #print(f'epoch [{epoch+1}/{num_epochs}], train_loss: g_loss: {g_loss.item()}, d_loss: {d_loss.item()}, val_loss: {val_loss / len(val_loader)}')
            f.write(f'epoch [{epoch+1}/{num_epochs}], train_loss: g_loss: {g_loss.item()}, d_loss: {d_loss.item()}, val_loss: {val_loss / len(val_loader)}\n')