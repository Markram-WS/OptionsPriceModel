from tensorflow import keras
import tensorflow as tf


# https://keras.io/examples/generative/wgan_gp/
class OptionChainGenerator(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        discriminator_extra_steps=3,
        gp_weight=(10.0,),
        gp_cap=None,
        output_col=[],
        scaler=[],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.gp_cap = gp_cap
        # ---other---
        self.output_col = output_col
        # zero values
        self.scaler = scaler

    def compile(self, d_optimizer, g_optimizer):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        self.generator_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.generator_loss_tracker_wgan = tf.keras.metrics.Mean(
            name="generator_loss_wgan"
        )
        self.generator_loss_tracker_c1 = tf.keras.metrics.Mean(name="generator_loss_c1")
        self.generator_loss_tracker_c2 = tf.keras.metrics.Mean(name="generator_loss_c2")
        self.generator_loss_tracker_p1 = tf.keras.metrics.Mean(name="generator_loss_p1")
        self.generator_loss_tracker_p2 = tf.keras.metrics.Mean(name="generator_loss_p2")
        self.discriminator_loss_tracker = tf.keras.metrics.Mean(
            name="discriminator_loss"
        )
        self.d_cost_tracker = tf.keras.metrics.Mean(name="d_cost")
        self.gradient_penalty_tracker = tf.keras.metrics.Mean(name="gradient_penalty")

    @property
    def metrics(self):
        return [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.gradient_penalty_tracker,
            self.d_cost_tracker,
        ]

    def _discriminator_loss(self, real_logits, fake_logits):
        real_loss = tf.reduce_mean(real_logits)
        fake_loss = tf.reduce_mean(fake_logits)
        return fake_loss - real_loss

        """
        #! fix zero value data
        zero data
        UNDERLYING_LAST	STRIKE	STRIKE_DISTANCE	INTRINSIC_VALUE	DTE	TOTAL_VOLUME	C_VEGA	P_VEGA	C_BID	C_ASK	C_VOLUME	P_BID	P_ASK	P_VOLUME
        1137.471494	1105.070607	78.562482	32.400888	160.424048	3096.693716	1.951961	-15.65377	81.27063	83.551341	86.733379	46.981467	48.860226	127.545349

        """

    # Define the loss functions for the generator.
    def _generator_loss(self, fake_data, generated_data):
        wgan_loss = -tf.reduce_mean(fake_data)
        # wgan_loss = 0.0
        # create data dict
        # Convert the tensor to a dictionary
        data_dict = {
            self.output_col[i]: generated_data[:, :, i]
            for i in range(len(self.output_col))
        }

        # [c1] additional condition loss  c_ask > c_bid
        additional_con_loss_c_1 = tf.reduce_mean(
            data_dict["C_ASK"] - data_dict["C_BID"]
        )
        # [c2] additional condition loss  call (s-x)
        # C_BID[0] : -4.999975e-14
        # C_ASK[0] : -4.999975e-14
        additional_con_loss_c_2 = []
        for c, v in [("C_BID", -4.999975e-14), ("C_ASK", -4.999975e-14)]:
            # tf mark with zero
            mask = tf.greater(data_dict[c], v)
            filtered_tensor = tf.boolean_mask(data_dict[c], mask)
            filtered_tensor_roll = tf.roll(filtered_tensor, shift=-1, axis=0)
            difference = tf.maximum(
                filtered_tensor_roll[:-1] - filtered_tensor[:-1], 0.0
            )
            additional_con_loss_c_2.append(tf.reduce_sum(difference))
        additional_con_loss_c_2 = tf.reduce_mean(additional_con_loss_c_2)

        # [p1] additional condition loss  p_ask > p_bid
        additional_con_loss_p_1 = tf.reduce_mean(
            data_dict["P_ASK"] - data_dict["P_BID"]
        )

        # [p2] additional condition loss  put (x-s)
        # P_BID[0] : -4.999975e-14
        # P_ASK[0] : -4.999975e-14
        additional_con_loss_p_2 = []
        for p, v in [("P_BID", -4.999975e-14), ("P_ASK", -4.999975e-14)]:
            # tf mark with zero
            mask = tf.greater(data_dict[p], v)
            filtered_tensor = tf.boolean_mask(data_dict[p], mask)
            filtered_tensor_roll = tf.roll(filtered_tensor, shift=-1, axis=0)
            difference = tf.maximum(
                filtered_tensor[:-1] - filtered_tensor_roll[:-1], 0.0
            )
            additional_con_loss_p_2.append(tf.reduce_sum(difference))
        additional_con_loss_p_2 = tf.reduce_mean(additional_con_loss_p_2)
        # additional condition zero var
        # DTE[0] : 160.424051
        # mask = tf.greater(mark with dit, v)

        return (
            wgan_loss,
            additional_con_loss_c_1 * 0.01,
            0,
            additional_con_loss_p_1 * 0.01,
            0,
        )

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):
        if isinstance(data, tuple):
            input_data, real_data = data

        # Get the batch size
        batch_size = tf.shape(real_data)[0]

        # zeroArr = np.zeros((1, 16, 14))
        # zeroArr=Scaler.transform(zeroArr.reshape(-1,(1, 16, 14)[-1]))
        # zeroArr = zeroArr.reshape(1,16,14)[:, :, select_y]

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
                d_cost = self._discriminator_loss(real_logits, fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_data, fake_data)
                if self.gp_cap:
                    gp = tf.minimum(gp, self.gp_cap)
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
            generated_data = self.generator(input_data, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_data, training=True)
            # Calculate the generator loss
            (
                wgan_loss,
                additional_con_loss_c_1,
                additional_con_loss_c_2,
                additional_con_loss_p_1,
                additional_con_loss_p_2,
            ) = self._generator_loss(gen_img_logits, generated_data)
            g_loss = (
                wgan_loss
                + additional_con_loss_c_1
                + additional_con_loss_c_2
                + additional_con_loss_p_1
                + additional_con_loss_p_2
            )

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        self.generator_loss_tracker.update_state(g_loss)
        self.generator_loss_tracker_wgan.update_state(wgan_loss)
        self.generator_loss_tracker_c1.update_state(additional_con_loss_c_1)
        self.generator_loss_tracker_c2.update_state(additional_con_loss_c_2)
        self.generator_loss_tracker_p1.update_state(additional_con_loss_p_1)
        self.generator_loss_tracker_p2.update_state(additional_con_loss_p_2)
        self.discriminator_loss_tracker.update_state(d_loss)
        self.gradient_penalty_tracker.update_state(gp)
        self.d_cost_tracker.update_state(d_cost)

        return {
            "generator_loss": self.generator_loss_tracker.result(),
            "generator_loss_wgan": self.generator_loss_tracker_wgan.result(),
            "generator_loss_c1": self.generator_loss_tracker_c1.result(),
            "generator_loss_c2": self.generator_loss_tracker_c2.result(),
            "generator_loss_p1": self.generator_loss_tracker_p1.result(),
            "generator_loss_p2": self.generator_loss_tracker_p2.result(),
            "discriminator_loss": self.discriminator_loss_tracker.result(),
            "gradient_penalty": self.gradient_penalty_tracker.result(),
            "d_cost": self.d_cost_tracker.result(),
        }

    def test_step(self, data):
        x_data, y_data = data
        gen_data = self.generator(x_data)
        # Get critic's predictions
        real_img_logits = self.discriminator(y_data, training=False)
        fake_img_logits = self.discriminator(gen_data, training=False)

        # Calculate loss
        discriminator_loss = tf.reduce_mean(real_img_logits) - tf.reduce_mean(
            fake_img_logits
        )
        generator_loss = -tf.reduce_mean(fake_img_logits)

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(discriminator_loss)

        return {
            "generator_loss": self.generator_loss_tracker.result(),
            "discriminator_loss": self.discriminator_loss_tracker.result(),
        }
