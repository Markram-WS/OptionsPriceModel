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

        self.gradient_penalty_tracker = tf.keras.metrics.Mean(name="gradient_penalty")
        self.fake_logits_tracker = tf.keras.metrics.Mean(name="fake_logits")
        self.real_logits_tracker = tf.keras.metrics.Mean(name="real_logits")
        self.diff_logits_tracker = tf.keras.metrics.Mean(name="diff_logits")

    @property
    def metrics(self):
        return [
            self.generator_loss_tracker,
            self.discriminator_loss_tracker,
            self.gradient_penalty_tracker,
            self.fake_logits_tracker,
            self.real_logits_tracker,
            self.diff_logits_tracker,
        ]

    def _discriminator_loss(self, real_logits, fake_logits):
        real_loss = tf.reduce_mean(real_logits)
        fake_loss = tf.reduce_mean(fake_logits)
        return fake_loss - real_loss

    # Define the loss functions for the generator.
    def _generator_loss(self, generated_data, real_data):
        wgan_loss = tf.reduce_mean(tf.square(generated_data - real_data))
        # wgan_loss = 0.0
        # create data dict
        # Convert the tensor to a dictionary
        data_dict = {
            self.output_col[i]: generated_data[:, :, i]
            for i in range(len(self.output_col))
        }

        # [c1] additional condition loss  c_ask > c_bid
        # Bid/Ask zero values : -0.432817
        additional_con_loss_c_1 = tf.reduce_mean(
            tf.square(tf.maximum(-(data_dict["C_BID"] + 0.432817), 0))
        )
        additional_con_loss_c_2 = tf.reduce_mean(
            tf.square(tf.maximum(-(data_dict["C_ASK"] + 0.432817), 0))
        )
        # [c2] additional condition loss  call (s-x)
        # C_BID[0] : -4.999975e-14
        # C_ASK[0] : -4.999975e-14
        # additional_con_loss_c_2 = []
        # for c, v in [("C_BID", -4.999975e-14), ("C_ASK", -4.999975e-14)]:
        #     # tf mark with zero
        #     mask = tf.greater(data_dict[c], v)
        #     filtered_tensor = tf.boolean_mask(data_dict[c], mask)
        #     filtered_tensor_roll = tf.roll(filtered_tensor, shift=-1, axis=0)
        #     difference = tf.maximum(
        #         filtered_tensor_roll[:-1] - filtered_tensor[:-1], -0.01
        #     )
        #     additional_con_loss_c_2.append(tf.reduce_sum(difference))
        # additional_con_loss_c_2 = tf.reduce_mean(additional_con_loss_c_2)

        # [p1] additional condition loss  p_ask > p_bid
        # Bid/Ask zero values : -0.432817
        additional_con_loss_p_1 = tf.reduce_mean(
            tf.square(tf.maximum(-(data_dict["P_ASK"] + 0.432817), 0))
        )
        additional_con_loss_p_2 = tf.reduce_mean(
            tf.square(tf.maximum(-(data_dict["P_BID"] + 0.432817), 0))
        )
        # [p2] additional condition loss  put (x-s)
        # P_BID[0] : -4.999975e-14
        # P_ASK[0] : -4.999975e-14
        # additional_con_loss_p_2 = []
        # for p, v in [("P_BID", -4.999975e-14), ("P_ASK", -4.999975e-14)]:
        #     # tf mark with zero
        #     mask = tf.greater(data_dict[p], v)
        #     filtered_tensor = tf.boolean_mask(data_dict[p], mask)
        #     filtered_tensor_roll = tf.roll(filtered_tensor, shift=-1, axis=0)
        #     difference = tf.maximum(
        #         filtered_tensor[:-1] - filtered_tensor_roll[:-1], 0.0
        #     )
        #     additional_con_loss_p_2.append(tf.reduce_sum(difference))
        # additional_con_loss_p_2 = tf.reduce_mean(additional_con_loss_p_2)
        # additional condition zero var
        # DTE[0] : 160.424051
        # mask = tf.greater(mark with dit, v)

        return (
            wgan_loss,
            additional_con_loss_c_1,
            additional_con_loss_c_2,
            additional_con_loss_p_1,
            additional_con_loss_p_2,
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

        ## Get the batch size
        # batch_size = tf.shape(real_data)[0]

        # Train the generator
        # Get the latent vector
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_data = self.generator(input_data, training=True)
            # Calculate the generator loss
            (
                wgan_loss,
                additional_con_loss_c_1,
                additional_con_loss_c_2,
                additional_con_loss_p_1,
                additional_con_loss_p_2,
            ) = self._generator_loss(generated_data, real_data)
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
        self.discriminator_loss_tracker.update_state(0)
        self.gradient_penalty_tracker.update_state(0)
        self.fake_logits_tracker.update_state(0)
        self.real_logits_tracker.update_state(0)
        self.diff_logits_tracker.update_state(0)

        return {
            "generator_loss": self.generator_loss_tracker.result(),
            "generator_loss_wgan": self.generator_loss_tracker_wgan.result(),
            "generator_loss_c1": self.generator_loss_tracker_c1.result(),
            "generator_loss_c2": self.generator_loss_tracker_c2.result(),
            "generator_loss_p1": self.generator_loss_tracker_p1.result(),
            "generator_loss_p2": self.generator_loss_tracker_p2.result(),
            "discriminator_loss": self.discriminator_loss_tracker.result(),
            "gradient_penalty": self.gradient_penalty_tracker.result(),
            "fake_logits": self.fake_logits_tracker.result(),
            "real_logits": self.real_logits_tracker.result(),
            "diff_logits": self.diff_logits_tracker.result(),
        }

    def test_step(self, data):
        x_data, y_data = data
        gen_data = self.generator(x_data)

        generator_loss = tf.reduce_mean(tf.square(gen_data - y_data))

        self.generator_loss_tracker.update_state(generator_loss)
        self.discriminator_loss_tracker.update_state(0)

        return {
            "generator_loss": self.generator_loss_tracker.result(),
            "discriminator_loss": self.discriminator_loss_tracker.result(),
        }
