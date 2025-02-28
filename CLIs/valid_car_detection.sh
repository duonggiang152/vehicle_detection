TEST_DATA_DIR='/home/giang/data/cardetection/data/training_images'
TEST_ANNOTATION_FILE_DIR='/home/giang/data/cardetection/data/coco_valid.json'
OUTPUT_TFRECORD_TEST='./../output/car_detection/valid'

python -m official.vision.data.create_coco_tf_record --logtostderr \
  --image_dir=$TEST_DATA_DIR \
  --object_annotations_file=$TEST_ANNOTATION_FILE_DIR \
  --output_file_prefix=$OUTPUT_TFRECORD_TEST \
  --num_shards=1