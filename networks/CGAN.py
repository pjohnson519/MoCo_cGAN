from keras.layers import Input, Lambda
from keras.models import Model


def CGAN(generator_model, discriminator_model, input_img_dim):
    """
    
    1. Generate an image with the generator
    
    2. feed the images to a discriminator 
    3. the CGAN outputs the generated image and the loss

    """

    generator_input = Input(shape=input_img_dim, name="CGAN_input")

    # generated image model from the generator
    generated_image = generator_model(generator_input)

    h= input_img_dim[1]
    w= input_img_dim[0]

            
    # measure loss from patches of the image (not the actual image)
    cgan_output = discriminator_model(generated_image)

    # actually turn into keras model
    cgan = Model(input=[generator_input], output=[generated_image, cgan_output], name="CGAN")
    return cgan
