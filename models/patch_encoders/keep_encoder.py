from transformers import AutoModel, AutoTokenizer
from torchvision import transforms
from PIL import Image

# model = AutoModel.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("Astaxanthin/KEEP", trust_remote_code=True)
# model.eval()
# transform = transforms.Compose([
#     transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.CenterCrop(size=(224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
# ])

# example_image_path = './example.tif'
# example_text = ['an H&E image of breast invasive carcinoma.', 'an H&E image of normal tissue.', 'an H&E image of lung adenocarcinoma.']

# img_input =  transform(Image.open(example_image_path).convert('RGB')).unsqueeze(0)
# token_input = tokenizer(example_text,max_length=256,padding='max_length',truncation=True, return_tensors='pt')

# img_feature = model.encode_image(img_input)
# text_feature = model.encode_text(token_input)