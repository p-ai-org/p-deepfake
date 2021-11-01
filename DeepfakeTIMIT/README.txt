
DeepfakeTIMIT is a database of videos where faces are swapped using the open source GAN-based approach (adapted from here: https://github.com/shaoanlu/faceswap-GAN), 
which, in turn, was developed from the original autoencoder-based Deepfake algorithm (https://github.com/deepfakes/faceswap). 

When creating the database, we manually selected 16 similar looking pairs of people from publicly available VidTIMIT database (http://conradsanderson.id.au/vidtimit/). For each of 32 subjects, we trained two different models: a lower quality (LQ) with 64 x 64 input/output size model, and higher quality (HQ) with 128 x 128 size model (see the available images for the illustration). Since there are 10 videos per person in VidTIMIT database, we generated 320 videos corresponding to each version, resulting in 620 total videos with faces swapped. For the audio, we kept the original audio track of each video, i.e., no manipulation was done to the audio channel.


Any publication (eg. conference paper, journal article, technical report, book chapter, etc) resulting from the usage of DeepfakeTIMIT must cite the following paper:

    P. Korshunov and S. Marcel,
    DeepFakes: a New Threat to Face Recognition? Assessment and Detection.
    arXiv and Idiap Research Report


Any publication (eg. conference paper, journal article, technical report, book chapter, etc) resulting from the usage of VidTIMIT and subsequently DeepfakeTIMIT must also cite the following paper:

    C. Sanderson and B.C. Lovell,
    Multi-Region Probabilistic Histograms for Robust and Scalable Identity Inference.
    Lecture Notes in Computer Science (LNCS), Vol. 5558, pp. 199-208, 2009.
