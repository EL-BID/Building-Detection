from os.path import join, abspath, dirname
import json

import rastervision as rv
from rastervision.utils.files import file_to_str

# These paths need to be set for your environment.
# The rv_root is where RV will put output.
remote_rv_root = 's3://azavea-idb-data/buildings/idb-data-bundle-2-19/rv'
local_rv_root = '/opt/data/rv'

# The data_root is where the `input` directory of the data bundle is located.
remote_data_root = 's3://azavea-idb-data/buildings/idb-data-bundle-2-19/input'
local_data_root = '/opt/data/input'

scenes_config_path = join(dirname(abspath(__file__)), 'scenes-config.json')


def build_scene(remote, test, task, scene_config, aoi_inds, channel_order):
    data_root = remote_data_root if remote else local_data_root
    vector_tile_zoom = 12
    class_id_to_filter = {1: ['has', 'building']}
    raster_uris = [join(data_root, i) for i in scene_config['images']]
    shifts = scene_config.get('shifts', [0, 0])

    raster_source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                      .with_uris(raster_uris) \
                      .with_channel_order(channel_order) \
                      .with_shifts(shifts[0], shifts[1]) \
                      .build()

    vector_tile_uri = join(data_root, scene_config['labels'])
    vector_source = rv.VectorSourceConfig.builder(rv.VECTOR_TILE_SOURCE) \
        .with_class_inference(class_id_to_filter=class_id_to_filter,
                              default_class_id=None) \
        .with_uri(vector_tile_uri) \
        .with_zoom(vector_tile_zoom) \
        .with_id_field('@id') \
        .build()

    background_class_id = 2
    label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
        .with_vector_source(vector_source) \
        .with_rasterizer_options(background_class_id) \
        .build()

    aoi_uris = [join(data_root, scene_config['aois'][aoi_ind]) for aoi_ind in aoi_inds]
    label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
        .with_raster_source(label_raster_source) \
        .build()

    vector_output = {'mode': 'polygons', 'class_id': 1, 'denoise': 5}
    label_store = rv.LabelStoreConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
                                     .with_vector_output([vector_output]) \
                                     .build()

    scene = rv.SceneConfig.builder() \
                          .with_task(task) \
                          .with_id(scene_config['id']) \
                          .with_raster_source(raster_source) \
                          .with_label_source(label_source) \
                          .with_label_store(label_store) \
                          .with_aoi_uris(aoi_uris) \
                          .build()

    return scene


def build_scenes(remote, test, task, channel_order):
    scenes_config = json.loads(file_to_str(scenes_config_path))
    train_scenes = []
    val_scenes = []

    if test:
        splits = {
            'paramaribo_test': {
                'train': [0],
                'test': [1]
            }
        }
    else:
        splits = {
            'belice': {
                'train': [0, 1],
                'test': [2]
            },
            'georgetown': {
                'train': [0, 1],
                'test': [4]
            },
            'paramaribo': {
                'train': [0, 1],
                'test': [2]
            }
        }

    for city, split in splits.items():
        if split.get('train'):
            scene = build_scene(remote, test, task, scenes_config[city], split['train'],
                                channel_order=channel_order)
            train_scenes.append(scene)

        if split.get('test'):
            scene = build_scene(remote, test, task, scenes_config[city], split['test'],
                                channel_order=channel_order)
            val_scenes.append(scene)

    return train_scenes, val_scenes


def str_to_bool(x):
    if type(x) == str:
        if x.lower() == 'true':
            return True
        elif x.lower() == 'false':
            return False
        else:
            raise ValueError('{} is expected to be true or false'.format(x))
    return x


class MultiCity(rv.ExperimentSet):
    def exp_main(self, test=False, remote=False):
        """Run an experiment on multiple cities.

        Args:
            test: (bool) if True, run a very small experiment as a test and generate
                debug output
            remote: (bool) if True, use remote URIs for data.
        """
        test = str_to_bool(test)
        remote = str_to_bool(remote)
        rv_root = remote_rv_root if remote else local_rv_root
        exp_id = 'multi-city'

        channel_order = [0, 1, 2]
        debug = False
        batch_size = 8
        num_steps = 150000
        model_type = rv.MOBILENET_V2

        if test:
            debug = True
            num_steps = 1
            batch_size = 1

        class_map = {
            'Building': (1, 'orange'),
            'Background': (2, 'black')
        }

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(300) \
                            .with_classes(class_map) \
                            .with_chip_options(
                                stride=150,
                                window_method='sliding',
                                debug_chip_probability=0.25) \
                            .build()

        backend = rv.BackendConfig.builder(rv.TF_DEEPLAB) \
                                  .with_task(task) \
                                  .with_model_defaults(model_type) \
                                  .with_config({
                                    'min_scale_factor': '0.75',
                                    'max_scale_factor': '1.25'},
                                    ignore_missing_keys=True, set_missing_keys=True) \
                                  .with_train_options(sync_interval=600) \
                                  .with_num_steps(num_steps) \
                                  .with_batch_size(batch_size) \
                                  .with_debug(debug) \
                                  .build()

        train_scenes, val_scenes = build_scenes(
            remote, test, task, channel_order)
        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(rv_root) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
