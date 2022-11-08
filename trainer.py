import torch
import torch.nn.functional as F
from tqdm import tqdm

from metric import count_FA_FR, get_au_fa_fr


def train_epoch(model, opt, loader, log_melspec, device):
    model.train()
    for i, (batch, labels) in tqdm(enumerate(loader), total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        opt.zero_grad()

        logits = model(batch)
        probs = F.softmax(logits, dim=-1)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        opt.step()

        argmax_probs = torch.argmax(probs, dim=-1)
        FA, FR = count_FA_FR(argmax_probs, labels)
        acc = torch.sum(argmax_probs == labels) / torch.numel(argmax_probs)

    return acc


def distill_train_epoch(
        teacher_model,
        student_model,
        student_opt,
        scheduler,
        loader,
        log_melspec,
        device,
        temperature: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.5,
):
    """
    Training the student model on a soft target distribution from the teacher model.
    """
    teacher_model.eval()
    student_model.train()

    for i, (batch, labels) in tqdm(enumerate(loader), total=len(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        student_opt.zero_grad()

        teacher_logits = teacher_model(batch)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        student_logits = student_model(batch)
        student_probs = F.softmax(student_logits, dim=-1)

        student_loss = F.cross_entropy(student_logits, labels)
        distill_loss = -1.0 * (teacher_probs * student_probs).sum(dim=1).mean()
        loss = alpha * student_loss + beta * distill_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 5)

        student_opt.step()
        if scheduler: scheduler.step()

        argmax_student_probs = torch.argmax(student_probs, dim=-1)
        FA, FR = count_FA_FR(argmax_student_probs, labels)
        acc = torch.sum(argmax_student_probs == labels) / torch.numel(argmax_student_probs)

    return acc


@torch.no_grad()
def validation(model, loader, log_melspec, device):
    model.eval()

    val_losses, accs, FAs, FRs = [], [], [], []
    all_probs, all_labels = [], []
    for i, (batch, labels) in tqdm(enumerate(loader)):
        batch, labels = batch.to(device), labels.to(device)
        batch = log_melspec(batch)

        output = model(batch)
        probs = F.softmax(output, dim=-1)
        loss = F.cross_entropy(output, labels)

        argmax_probs = torch.argmax(probs, dim=-1)
        all_probs.append(probs[:, 1].cpu())
        all_labels.append(labels.cpu())
        val_losses.append(loss.item())
        accs.append(
            torch.sum(argmax_probs == labels).item() /  # ???
            torch.numel(argmax_probs)
        )
        FA, FR = count_FA_FR(argmax_probs, labels)
        FAs.append(FA)
        FRs.append(FR)

    au_fa_fr = get_au_fa_fr(torch.cat(all_probs, dim=0).cpu(), all_labels)
    return au_fa_fr
