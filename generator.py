import torch
import torch.nn as nn



class BuildConv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=True, activation=True):
        super(BuildConv, self).__init__()
        self.Conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.batch_norm = batch_norm
        self.BatchNorm = nn.BatchNorm2d(output_size)

    def forward(self, x):
        if self.activation:
            op = self.Conv(self.leaky_relu(x))
        else:
            op = self.Conv(x)

        if self.batch_norm:
            return self.BatchNorm(op)
        else:
            return op

class BuildDeconv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2,padding=1, batch_norm=True,dropout=False):
        super(BuildDeconv, self).__init__()
        self.DeConv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.BatchNorm = nn.BatchNorm2d(output_size)
        self.Dropout = nn.Dropout2d(0.5)
        self.relu = nn.ReLU(True)
        self.dropout=dropout
        self.batch_norm=batch_norm

    def forward(self, x):
        if self.batch_norm:
            op = self.BatchNorm(self.DeConv(self.relu(x)))
        else:
            op = self.DeConv(self.relu(x))

        if self.dropout:
            return self.Dropout(op)
        else:
            return op

class BuildUpsample(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2,padding=1, batch_norm=True,dropout=False):
        super(BuildUpsample, self).__init__()
        self.PixelShuffle = nn.PixelShuffle(2)
        self.BatchNorm = nn.BatchNorm2d(output_size)
        self.Dropout = nn.Dropout2d(0.5)
        self.relu = nn.ReLU(True)
        self.dropout=dropout
        self.batch_norm=batch_norm

    def forward(self, x):
        if self.batch_norm:
            op = self.BatchNorm(self.PixelShuffle(self.relu(x)))
        else:
            op = self.PixelShuffle(self.relu(x))

        if self.dropout:
            return self.Dropout(op)
        else:
            return op


class Generator(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim):
        super(Generator, self).__init__()

        # Encoder
        # self.conv1
        self.conv1 = BuildConv(input_dim, num_filters, batch_norm=True, activation=False)
        self.conv2 = BuildConv(num_filters, num_filters * 2)
        self.conv3 = BuildConv(num_filters * 2, num_filters * 4)
        self.conv4 = BuildConv(num_filters * 4, num_filters * 8)
        self.conv5 = BuildConv(num_filters * 8, num_filters * 8)
        self.conv6 = BuildConv(num_filters * 8, num_filters * 8)
        self.conv7 = BuildConv(num_filters * 8, num_filters * 8)
        self.conv8 = BuildConv(num_filters * 8, num_filters * 8, batch_norm=False)

        # Decoder

        self.deconv1 = BuildDeconv(num_filters * 8, num_filters * 8, dropout=True)
        self.deconv2 = BuildDeconv(num_filters * 8 * 2, num_filters * 8, dropout=True)
        self.deconv3 = BuildDeconv(num_filters * 8 * 2, num_filters * 8, dropout=True)
        self.deconv4 = BuildDeconv(num_filters * 8 * 2, num_filters * 8)
        self.deconv5 = BuildDeconv(num_filters * 8 * 2, num_filters * 4)
        self.deconv6 = BuildDeconv(num_filters * 4 * 2, num_filters * 2)
        self.deconv7 = BuildDeconv(num_filters * 2 * 2, num_filters)
        self.deconv8 = BuildDeconv(num_filters * 2, output_dim, batch_norm=False)

    def forward(self, x):

        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        enc7 = self.conv7(enc6)
        enc8 = self.conv8(enc7)

        # Decoder with skip connections
        dec1 = self.deconv1(enc8)
        dec1 = torch.cat([dec1, enc7], 1)
        dec2 = self.deconv2(dec1)
        dec2 = torch.cat([dec2, enc6], 1)
        dec3 = self.deconv3(dec2)
        dec3 = torch.cat([dec3, enc5], 1)
        dec4 = self.deconv4(dec3)
        dec4 = torch.cat([dec4, enc4], 1)
        dec5 = self.deconv5(dec4)
        dec5 = torch.cat([dec5, enc3], 1)
        dec6 = self.deconv6(dec5)
        dec6 = torch.cat([dec6, enc2], 1)
        dec7 = self.deconv7(dec6)
        dec7 = torch.cat([dec7, enc1], 1)
        dec8 = self.deconv8(dec7)
        out = torch.nn.Tanh()(dec8)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        ''' This function initializes weights of layers'''
        for m in self.children():

            if isinstance(m, BuildConv):
                nn.init.normal_(m.Conv.weight, mean, std)

            if isinstance(m, BuildDeconv):
                nn.init.normal_(m.DeConv.weight, mean, std)


class OpsAdvGenerator(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim):
        super(OpsAdvGenerator, self).__init__()

        # Encoder
        self.conv1 = BuildConv(input_dim, num_filters, batch_norm=True, activation=False)
        self.conv2 = BuildConv(num_filters, num_filters*2)
        self.conv3 = BuildConv(num_filters*2, num_filters*4)
        self.conv4 = BuildConv(num_filters*4, num_filters*8)

        # Decoder
        self.deconv1 = BuildDeconv(num_filters*8, num_filters*4, dropout=True)
        self.deconv2 = BuildDeconv(num_filters*4*2, num_filters*2)
        self.deconv3 = BuildDeconv(num_filters*2*2, num_filters)
        self.deconv4 = BuildDeconv(num_filters*2, output_dim, batch_norm=False)

    def forward(self, x):

        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)


        # Decoder with skip connections
        dec1 = self.deconv1(enc4)
        dec1 = torch.cat([dec1, enc3], dim=1)
        dec2 = self.deconv2(dec1)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec3 = self.deconv3(dec2)
        dec3 = torch.cat([dec3, enc1], dim=1)
        dec4 = self.deconv4(dec3)

        output = torch.nn.Tanh()(dec4)

        return output

    def normal_weight_init(self, mean=0.0, std=0.02):
        """this function initializes weights of layers"""
        for m in self.children():
            if isinstance(m, BuildConv):
                nn.init.normal_(m.Conv.weight, mean, std)
            if isinstance(m, BuildDeconv):
                nn.init.normal_(m.DeConv.weight, mean, std)


class OpsAdvGenerator_PixelShuffle(nn.Module):
    def __init__(self, input_dim, num_filters, output_dim):
        super(OpsAdvGenerator_PixelShuffle, self).__init__()

        # Encoder
        self.conv1 = BuildConv(input_dim, num_filters, batch_norm=True, activation=False)
        self.conv2 = BuildConv(num_filters, num_filters*2)
        self.conv3 = BuildConv(num_filters*2, num_filters*4)
        self.conv4 = BuildConv(num_filters*4, num_filters*16)

        # Decoder
        self.deconv1 = BuildUpsample(num_filters*8, num_filters*4, dropout=True)
        self.deconv2 = BuildUpsample(num_filters*4*2, num_filters*2)
        self.deconv3 = BuildUpsample(num_filters*2*2, num_filters)
        self.deconv4 = BuildUpsample(num_filters*2, output_dim, batch_norm=False)

        self.conv_final = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1, stride=1)

    def forward(self, x):

        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)


        # Decoder with skip connections
        dec1 = self.deconv1(enc4)
        dec1 = torch.cat([dec1, enc3], dim=1)
        dec2 = self.deconv2(dec1)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec3 = self.deconv3(dec2)
        dec3 = torch.cat([dec3, enc1], dim=1)
        dec4 = self.deconv4(dec3)

        dec4_final = self.conv_final(dec4)
        output = torch.nn.Tanh()(dec4_final)

        return output

    def normal_weight_init(self, mean=0.0, std=0.02):
        """this function initializes weights of layers"""
        for m in self.children():
            if isinstance(m, BuildConv):
                nn.init.normal_(m.Conv.weight, mean, std)
            if isinstance(m, BuildDeconv):
                nn.init.normal_(m.DeConv.weight, mean, std)




class CGenerator(nn.Module):
    # def __init__(self, in_ch, out_ch, n_classes, latent_dim=128, img_shape=224):
    def __init__(self, n_classes=1000, latent_dim=128, img_shape=(3, 224,224)):
        super(CGenerator, self).__init__()
        self.img_shape = img_shape
        self.cls_embed = nn.Linear(n_classes, latent_dim)
        # self.pre_train = nn.Sequential(models.vgg16(pretrained=True).features)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # *block(latent_dim + n_classes, 128, normalize=False),
            *block(latent_dim * 2, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )


    def forward(self, z_s, labels):
        gen_input = torch.cat((z_s,self.cls_embed(labels)), dim=-1)
        img = self.model(gen_input)
        img = img.view(img.size(0), self.img_shape[0], self.img_shape[1],self.img_shape[2])
        return img
