from model.DABNet import DABNet


def build_model(model_name, num_classes):
    if model_name == 'DABNet':
        return DABNet(classes=num_classes)
