# Introduction
This project aim to verify the rather complicated captcha while signing into a government website (Not to specify which government here)

Difficulties in verifying this captcha
- Images with various size
- The dots in the background usually are not single dots but a cluster of dots
- Lots of different font and colors in the characters of the Captcha
- A large straight line striking through the characters in the Captcha
- Numerous small thin lines distributed randomly in the Captcha, some of which can be the same thickness as the characters themselves

To solve this Captcha, I used the open cv to remove some of the noises first and feed them into a Keras machine learning model for it to recognise the pattern.

# Open CV:
![Original Images](https://github.com/matthewmakhl/verify-captcha/tree/master/README_images/1.png?raw=true)
1. I changed all the grey-ish color points into white color to remove 80% of the dots. I tried to remove this using Open CV's threshold function but it turned out that it will make the character itself to be thin out and harder for the Keras model to recognise. Using color range to filter is more effective as the majority of the characters are not in grey color thus not affecting the characters themselves
![First changes](https://github.com/matthewmakhl/verify-captcha/tree/master/README_images/2.png?raw=true)
1. Then I changed the 3 color RGB channel Captcha images into black and white 1 channel images. This effectively negate the effect of having characters with different colors
![Second changes](https://github.com/matthewmakhl/verify-captcha/tree/master/README_images/3.png?raw=true)
1. As Keras model works better on fixed size images, I checked the max height and width in my images sample and fill every images with white space if they are not as large as that size. I don't want to simply enlarge the images to that size as that may lower the image quality and distort the character
![Third changes](https://github.com/matthewmakhl/verify-captcha/tree/master/README_images/4.png?raw=true)
1. Then I used the Open CV's threshold function to remove the remainly dots and thin line. I used trial and error to test the most effective configuration and make it just enough to remove most noises and not to thin out the characters themselves too much
![Final result](https://github.com/matthewmakhl/verify-captcha/tree/master/README_images/5.png?raw=true)
1. I tried to detect the large straight line in the Captcha and use inverse mask to remove it using Open CV's hough line function. During my testing, it requires around 3px thick lines to mask it out. which also cross out quite a large portion of the characters. After testing on both the masked and unmasked version of Captcha on the Keras model, it seems the Keras model has a better learning rate and smaller loss for the unmasked version. So the final version didn't mask out the large straight line

# Keras Model:
I found many Keras model online and eventually came into a Keras model that was originally used to solve a simplier Captcha (Reference below). After briefly reviewing the codes, I found it also useful in my case.

I changed some of the variables and feed my training set of Captcha images to it (1000 images in the final version)

The current version have 7-80% accuracy on identifying individual characters, but much lower chance to predict all the characters in each captcha. Hopefully I can increase its accuracy further by providing more training sample