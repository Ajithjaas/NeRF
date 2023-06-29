import numpy as np
import matplotlib.pyplot as plt
import pry
import torch

from Network import NeRF
from Utilities.PositionalEncoding import positional_encoding


class Train:
    def __init__(self, height, width, images, poses, focal_length, camera_view_bounds):
        self.height = height
        self.width = width
        self.images = images
        self.poses = poses
        self.focalLength = focal_length
        self.camera_view_bounds = camera_view_bounds

    def get_rays(self, pose, ray_samples):
        x, y = torch.meshgrid(torch.arange(self.width).to(pose), torch.arange(self.height).to(pose))
        x, y = x.transpose(-1, -2), y.transpose(-1, -2)
        directions = torch.stack([(x - self.width * 0.5) / self.focalLength, -(y - self.height * 0.5) / self.focalLength,
                                  -torch.ones_like(x)], dim=-1)
        ray_directions = torch.sum(directions[..., None, :] * pose[:3, :3], dim=-1)
        ray_origins = pose[:3, -1].expand(ray_directions.shape)

        near = self.camera_view_bounds["near"]
        far = self.camera_view_bounds["far"]

        # Generate values
        sample_points = torch.linspace(near, far, ray_samples).to(ray_origins)

        noise_shape = list(ray_origins.shape[:-1]) + [ray_samples]

        sample_points = sample_points + torch.rand(noise_shape).to(ray_origins) * (far - near) / ray_samples

        query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * sample_points[..., :, None]

        return ray_origins, ray_directions, query_points, sample_points

    def render_volume_density(self, radiance_field, ray_origins, sample_points):
        sigma = torch.nn.functional.relu(radiance_field[..., 3])

        rgb = torch.sigmoid(radiance_field[..., :3])

        oneE10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)

        delta = torch.cat((sample_points[..., 1:] - sample_points[..., :-1], oneE10.expand(sample_points[..., :1].shape)), dim=-1)

        alpha = 1. - torch.exp(-sigma*delta)

        exponential_term = 1. - alpha
        epsilon = 1e-10
        transmittance = torch.roll(torch.cumprod(exponential_term + epsilon, -1), 1, -1)
        transmittance[..., 0] = 1

        weights = alpha * transmittance

        rgb_map = (weights[..., None]*rgb).sum(dim=-2)
        depth = (weights * sample_points).sum(dim=-1)

        weights = weights.sum(-1)

        return rgb_map, depth, weights

    def run_nerf_model(self, ray_samples, encode, pose, chunk_size, model):
        ray_origins, ray_directions, query_points, sample_points = self.get_rays(pose, ray_samples)

        flattened_query_points = query_points.reshape((-1, 3))
        # print("Flat query", flattened_query_points)
        encoded_points = encode(flattened_query_points)

        batches = [encoded_points[i:i + chunk_size] for i in range(0, encoded_points.shape[0], chunk_size)]

        predictions = []

        for batch in batches:
            predictions.append(model(batch))

        radiance_field_flattened = torch.cat(predictions, dim=0)

        unflattened_shape = list(query_points.shape[:-1]) + [4]

        radianceField = torch.reshape(radiance_field_flattened, unflattened_shape)

        rgb_predicted, depthPredicted, accuracy = self.render_volume_density(radianceField, ray_origins, sample_points)

        return rgb_predicted, depthPredicted, accuracy

    def train(self, num_encoding_functions, ray_samples, chunksize):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lr = 1e-2

        num_iterations = 50

        encode = lambda x: positional_encoding(x, num_encoding_functions)
        model = NeRF(encoding=num_encoding_functions)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        seed = 2
        torch.manual_seed(seed)
        np.random.seed(seed)

        iters = []
        log_loss_list = []

        for i in range(num_iterations):
            print("Iteration number: ", i)
            random_image_index = np.random.randint(self.images.shape[0])
            random_image = self.images[random_image_index].to(device)
            pose = self.poses[random_image_index].to(device)

            rgbPredicted, depthMap, accuracyMap = self.run_nerf_model(ray_samples, encode, pose, chunksize, model)

            # Get the prediction_loss between original and generated image
            random_image = random_image.float()
            prediction_loss = torch.nn.functional.mse_loss(rgbPredicted, random_image)

            print("Prediction loss:", prediction_loss.item())
            prediction_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Test for every 200 image
            if i % 200 == 0:
                test_image = self.images[-1].to(device)
                test_pose = self.poses[-1].to(device)

                test_rgbPredicted, _, _ = self.run_nerf_model(ray_samples, encode, test_pose, chunksize, model)
                prediction_loss = torch.nn.functional.mse_loss(test_rgbPredicted, test_image)
                print("prediction_loss:", prediction_loss.item())
                log_loss = -10. * torch.log10(prediction_loss)

                iters.append(i)
                log_loss_list.append(log_loss.detach().cpu().numpy())

        plt.figure(figsize=(10, 4))
        # plt.subplot(121)
        # plt.imshow(test_rgbPredicted.detach().cpu().numpy())
        # plt.title(f'iteartions{i}')
        # plt.subplot(122)
        plt.plot(iters, log_loss_list)
        plt.title("Log loss")
        plt.show()

