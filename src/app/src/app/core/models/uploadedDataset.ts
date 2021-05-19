import {NumberAttribute} from "./numberAttribute";
import {CategoricalAttribute} from "./categoricalAttribute";

export class UploadedDataset {
  name: string = '';
  rowsData: any;
  numberAttributes?: NumberAttribute[];
  categoricalAttributes?: CategoricalAttribute[];
}
