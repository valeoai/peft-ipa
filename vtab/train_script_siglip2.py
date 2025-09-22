import os, argparse
import torch
from transformers import (
    AutoProcessor,
    AutoModel,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score
import utils
import peft
import torch.utils.data as data
from tqdm import tqdm
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        return img, target

    def __len__(self):
        return len(self.imlist)

def get_raw_vtab_datasets(basedir, name):
    root = os.path.join(basedir, name)
    train_dataset = ImageFilelist(root=root, flist=root + "/train800val200.txt")
    val_dataset = ImageFilelist(root=root, flist=root + "/test.txt")
    return train_dataset, val_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       type=str, required=True)
    parser.add_argument("--data_path",     type=str, required=True)
    parser.add_argument("--exp_base_path", type=str, required=True)
    parser.add_argument("--batch_size",    type=int, default=64)
    parser.add_argument("--epochs",        type=int, default=100)
    parser.add_argument("--peft_model",    type=str, choices=["lora","dora","ipa"], required=True)
    parser.add_argument("--ipa_mode",      type=str, default="pre_ipca_requires_grad", choices=["pre_ipca_requires_grad","pre_ipca","pre_hebbian_requires_grad","pre_hebbian"])
    args = parser.parse_args()

    utils.mkdirss(args.exp_base_path)
    logger = utils.create_logger(args.exp_base_path, f"siglip2_{args.peft_model}_{args.dataset}")

    with open(f"./configs/{args.dataset}.txt", "r") as f:
        labels = [l.strip() for l in f if l.strip()]

    checkpoint = "google/siglip2-base-patch16-224"
    processor  = AutoProcessor.from_pretrained(checkpoint)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model      =  AutoModel.from_pretrained(checkpoint).to(device)

    template = "a photo of a {}"
    texts = [template.format(label.replace('_', ' ').lower()) for label in labels]
    processed_texts = processor(text=texts, padding='max_length', max_length=64, return_tensors="pt")

    if args.peft_model == "lora":
        peft_cfg = peft.LoraConfig(
            r=8, lora_alpha=8, lora_dropout=0.1,
            target_modules=r"^vision.*(q_proj|v_proj)$",
        )
    elif args.peft_model == "dora":
        peft_cfg = peft.LoraConfig(
            r=8, lora_alpha=8, lora_dropout=0.1,
            use_dora=True,
            target_modules=r"^vision.*(q_proj|v_proj)$",
        )
    elif args.peft_model == "ipa":
        ipa_mode = args.ipa_mode  # "pre_ipca" / "pre_ipca_requires_grad"
        peft_cfg = peft.IPAConfig(
            r=8, scaling=0.05, ipa_mode=ipa_mode, ipa_dropout=0.1,
            target_modules=r"^vision.*(q_proj|v_proj)$",
        )

    peft_model = peft.get_peft_model(model, peft_cfg).to(device)

    train_ds, val_ds = get_raw_vtab_datasets(
        basedir=args.data_path,
        name=args.dataset,
    )

    def collate_fn(batch):
        images, lbls = zip(*batch)
        img_batch = processor(images=list(images), return_tensors="pt")
        return {
            "pixel_values":    img_batch.pixel_values,
            "input_ids":       processed_texts["input_ids"].unsqueeze(0),
            "labels":          torch.tensor(lbls, dtype=torch.long),
        }

    class SiglipTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(
                pixel_values   = inputs["pixel_values"].to(device),
                input_ids      = inputs["input_ids"],
                return_dict    = True,
            )
            logits_per_image = outputs.logits_per_image
            int_labels = inputs["labels"]
            loss = torch.nn.functional.cross_entropy(logits_per_image, int_labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_preds):
        label_ids = eval_preds.label_ids
        cosine_similarity = eval_preds.predictions[0] 
        preds = cosine_similarity.argmax(axis=1)
        return {"val_acc": accuracy_score(label_ids, preds)}

    training_args = TrainingArguments(
        output_dir            = os.path.join(args.exp_base_path, f"siglip2_{args.peft_model}_{args.dataset}"),
        label_names           = ["labels"],
        num_train_epochs      = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        eval_accumulation_steps = 200,
        eval_strategy         = "steps",
        eval_steps            = 160,
        save_steps            = 160,
        logging_steps         = 160, 
        save_strategy         = "steps",
        learning_rate         = 1e-3,
        load_best_model_at_end= True,
        metric_for_best_model = "val_acc",
        report_to             = "none"
    )

    trainer = SiglipTrainer(
        model           = peft_model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        data_collator   = collate_fn,
        tokenizer       = processor,
        compute_metrics = compute_metrics,
    )

    if args.peft_model == 'ipa' and 'pre' in ipa_mode:
        print("Input projection pretraining......")
        dataloader = trainer.get_eval_dataloader(train_ds)

        with peft_model.disable_adapter():
            with torch.no_grad():
                peft_model.eval()
                for batch in tqdm(dataloader):
                    batch.pop("labels")
                    peft_model(**batch)

    peft_model.train()
    trainer.train()
    peft_model.eval()
    res = trainer.evaluate()

    logger.info(str(res))

if __name__ == "__main__":
    main()