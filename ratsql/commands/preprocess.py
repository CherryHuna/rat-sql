import argparse
import json

import _jsonnet
import tqdm
# from tqdm import tqdm
import sqlite3
from pathlib import Path

# These imports are needed for registry.lookup
# noinspection PyUnresolvedReferences
from ratsql import datasets
# noinspection PyUnresolvedReferences
from ratsql import grammars
# noinspection PyUnresolvedReferences
from ratsql import models
from ratsql.datasets.spider import SpiderItem
# noinspection PyUnresolvedReferences
from ratsql.utils import registry
# noinspection PyUnresolvedReferences
from ratsql.utils import vocab

from ratsql.datasets import spider

class Preprocessor:
    def __init__(self, config):
        self.eval_foreign_key_maps = None
        self.schemas = None
        self.config = config
        self.model_preproc = registry.instantiate(
            registry.lookup('model', config['model']).Preproc,
            config['model'])

    def preprocess(self):
        self.model_preproc.clear_items()
        for section in self.config['data']:
            data = registry.construct('dataset', self.config['data'][section])
            for item in tqdm.tqdm(data, desc=f"{section} section", dynamic_ncols=True):
                to_add, validation_info = self.model_preproc.validate_item(item, section)
                if to_add:
                    self.model_preproc.add_item(item, section, validation_info)
                if section == 'val':
                    break
        self.model_preproc.save()

    def preprocess_all(self):
        self.schemas, self.eval_foreign_key_maps = spider.load_tables(["./data/spider/tables.json"])

        for db_id, schema in tqdm.tqdm(self.schemas.items(), desc="DB connections"):
            sqlite_path = Path("./data/spider/database") / db_id / f"{db_id}.sqlite"
            source: sqlite3.Connection
            with sqlite3.connect(str(sqlite_path)) as source:
                dest = sqlite3.connect(':memory:')
                dest.row_factory = sqlite3.Row
                source.backup(dest)
            schema.connection = dest

    def preprocess_test(self, entry):
        # self.model_preproc.clear_items()， 这里清楚 val就行

        item = SpiderItem(
            text=entry['question_toks'],
            code=entry['sql'],
            schema=self.schemas[entry['db_id']],
            orig=entry,
            orig_schema=self.schemas[entry['db_id']].orig)
        to_add, validation_info = self.model_preproc.validate_item(item, 'val')
        if to_add:
            self.model_preproc.add_item(item, 'val', validation_info)

        self.model_preproc.save_pred()

def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-args')
    args = parser.parse_args()
    return args


def main(args):
    if args.config_args:
        config = json.loads(_jsonnet.evaluate_file(args.config, tla_codes={'args': args.config_args}))
    else:
        config = json.loads(_jsonnet.evaluate_file(args.config))

    preprocessor = Preprocessor(config)
    preprocessor.preprocess()


if __name__ == '__main__':
    args = add_parser()
    main(args)