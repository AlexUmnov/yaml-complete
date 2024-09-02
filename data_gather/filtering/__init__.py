from data_gather.filtering.simple import NonEmptyFilter, ValidYamlFilter

filter_registry = {
    "non_empty": NonEmptyFilter(),
    "valid_yaml": ValidYamlFilter()
}

