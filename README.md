This is a deep learning project which has created under the guidance
of Dr. Yedid Hoshen at the Hebrew Univeristy.
The goal of the project was to take the NAM algorithm from the repository
https://github.com/facebookresearch/NAM and to examine if another nets architectures
makes the results better.
The architectures which were examined are:
1. DiscoGAN from https://github.com/SKTBrain/DiscoGAN
2. UNIT from https://github.com/mingyuliutw/UNIT
3. UNET from https://github.com/milesial/Pytorch-UNet
From each of above I took the net architecture and I replaced the architecture
in the file conditional.py in the NAM project. Then I trained the program 
and tested the trained net on multiple images using the eval_variation.py code.
The results were written into the nam_eval_ims folder and were tested by visual assessment.

USAGE:
1. Download the Edges2Shoes data using the script get_data.sh in data folder.
This script also processes the data using the process_data.py file.
The default is 64*64 output images. If you want another image dimensions you should
pass it as an argument.
2. Train DCGAN unconditional generative model for the A domain using the train_gen.py
file in the code folder.
3. Choose one architecture from the folder code/newArchitectures
and replace the file conditional.py in the code folder with the chosen architecture.
4. Use NAM to train a mapping from A to B using the file train_nam.py (in the code folder).
5. Evaluate the trained net using the eval_variation.py file (in the code folder). 
If you want to test it on a specific image then add the image id as an argument.
The default is to test it on multiple images.
 
