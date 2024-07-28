from tensorflow import keras
import tensorflow as tf


# https://keras.io/examples/generative/wgan_gp/
class OptionChainGenerator(keras.Model):
    def __init__(self, discriminator, generator, **kwargs):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator

        self.d_steps = kwargs.get("discriminator_extra_steps", (3,))
        self.gp_weight = kwargs.get("gp_weight", (10.0,))

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.generator_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.discriminator_loss_tracker = tf.keras.metrics.Mean(
            name="discriminator_loss"
        )

    @property
    def metrics(self):
        return [self.generator_loss_tracker, self.discriminator_loss_tracker]

    def _discriminator_loss(real_data, fake_data):
        real_loss = tf.reduce_mean(real_data)
        fake_loss = tf.reduce_mean(fake_data)
        return fake_loss - real_loss

    # Define the loss functions for the generator.
    def _generator_loss(fake_data):
        return -tf.reduce_mean(fake_data)

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        if isinstance(data, tuple):
            input_data, real_data = data

        # Get the batch size
        batch_size = tf.shape(real_data)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for _ in range(self.d_steps):
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_data = self.generator(input_data, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_data, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_data, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self._discriminator_loss(
                    real_img=real_logits, fake_img=fake_logits
                )
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_data, fake_data)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(input_data, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self._generator_loss(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        self.generator_loss_tracker.update_state(g_loss)
        self.discriminator_loss_tracker.update_state(d_loss)

        return {
            "generator_loss": self.generator_loss_tracker.result(),
            "discriminator_loss": self.discriminator_loss_tracker.result(),
        }

    def test_step(self, data):
        x_data, y_data = data
        gen_data = self.generator(x_data)
        # Get critic's predictions
        real_output = self.discriminator(y_data, training=False)
        fake_output = self.discriminator(gen_data, training=False)

        # Calculate loss
        discriminator_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)
        generator_loss = -tf.reduce_mean(fake_output)

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)

        return {
            "generator_loss": self.generator_loss_tracker.result(),
            "discriminator_loss": self.discriminator_loss_tracker.result(),
        }
