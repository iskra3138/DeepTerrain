import torch

from dataloaders.NB_dataset import NBDataset
from dataloaders.NBYJ_dataset import NBYJDataset
from dataloaders.DALES_dataset import DALESDataset
from dataloaders.DALESDSP_dataset import DALESDSPDataset
from dataloaders.SUI_dataset import SUIDataset
from dataloaders.SUII_dataset import SUIIDataset
from dataloaders.KW_dataset import KWDataset
from dataloaders.RT_dataset import RTDataset
from dataloaders.DSM2DTM_dataset import DSM2DTMDataset

ALS2DTMDataset = {
    'DALES': DALESDataset,
    'NB': NBDataset,
    'NBYJ': NBYJDataset,
    'DALESDSP': DALESDSPDataset,
    'SUI': DSM2DTMDataset,
    'SUII': DSM2DTMDataset,
    'KW': DSM2DTMDataset,
    'RT': DSM2DTMDataset,
    'DSM2DTM': DSM2DTMDataset,
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
