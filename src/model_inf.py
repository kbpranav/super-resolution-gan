from models import Generator, ExtractFeatures, Discriminator
import torch

input_data = torch.randn(1 ,3, 224, 224)
input_gen = torch.randn(1,3,56,56)
# Generator
gen = Generator()
gen_out = gen.forward(input_gen)

print("Generator shape: ", gen_out.shape)

# ExtractFeatures
ext = ExtractFeatures()
ext_out = ext.forward(input_data)
print("Feature Extractor: ", ext_out.shape)

# Discriminator
disc = Discriminator(input_shape=(3, 224, 224))
disc_out = disc.forward(input_data)
print("Discriminator: ", disc_out.shape)

