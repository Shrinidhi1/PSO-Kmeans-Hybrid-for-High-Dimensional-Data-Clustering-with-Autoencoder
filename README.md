# PSO-Kmeans Hybrid for High Dimensional Data Clustering with Autoencoder
1. Design and develop a PSO algorithm for automatic data clustering.
2. Design and develop PSO employing Autoencoder for data clustering.
3. Compare the performance of PSO and Autoencoder based PSO data clustering algorithms using different validity indices.
4. Apply this algorithm on Stock Market Data and obtain inferences.

## Methodology
![System Design](https://github.com/Shrinidhi1/Stock-Market-Trends-using-PSO-Kmeans-Hybrid-Clustering-with-Autoencoder/assets/83594754/bf5972d2-72e7-4b7c-9f60-9d473c9396ba)

## Results

| Method | K-Means PSO     | K-Means PSO with Autoencoders |
| -------- | --------------------- | --------------------- |
| Dataset  | DB Index&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Silhouette Index| DB Index&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Silhouette Index|
| High     | 0.99316&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.044056| 0.499879&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.598376|
| Low      | 0.98635&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.079333| 0.492837&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.694484|
| Close    | 0.98474&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.046373| 0.474543&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.634368|
| Open     | 0.93643&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.056383| 0.547732&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.745483|
| Volume   | 0.99736&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.043367| 0.498746&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.648464|

A lower value of the DB Index and a higher value of the Silhouette Index indicate improved clustering performance.

## Conclusion
In terms of clustering accuracy and efficiency, the suggested method of utilizing PSO and K-Means with autoencoders has produced encouraging results. Our study demonstrates that, when applied to benchmark datasets like Nifty 100, our method outperforms other ones already in use. Investigating different Autoencoder variants for dimensionality reduction is one possible future direction. Applying this PSO and K-means algorithm to other, larger datasets is a different future approach that may be taken. It can be fascinating to apply the suggested method to practical applications in many industries, such as healthcare, banking, and image and video processing.
