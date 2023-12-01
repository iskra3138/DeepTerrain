import torch

from dataloaders.NB_dataset import NBDataset
from dataloaders.NBYJ_dataset import NBYJDataset
from dataloaders.DALES_dataset import DALESDataset
from dataloaders.DALESDSP_dataset import DALESDSPDataset

ALS2DTMDataset = {
    'DALES': DALESDataset,
    'NB': NBDataset,
    'NBYJ': NBYJDataset,
    'DALESDSP': DALESDSPDataset
}

def build_dataloader(split,
                     dataset_name,
                     batch_size,
                     num_workers,
                     logger,
                     shuffle=True,
                     pin_momory=True,
                     sampler=None,
                     ) :

    data_key = dataset_name
    dataset = ALS2DTMDataset[data_key](split=split,
                                       dataset_name=dataset_name)

    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               pin_memory=pin_momory,
                                               sampler=sampler)

    #print("\t==> data_loader size:{}".format(len(data_loader)))
    logger.info("\t==> data_loader size:{}".format(len(data_loader)))

    return data_loader, dataset.input_nc
