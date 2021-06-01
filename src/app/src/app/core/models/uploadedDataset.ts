import {NumberAttribute} from "./numberAttribute";
import {CategoricalAttribute} from "./categoricalAttribute";

export class UploadedDataset {
  id: number = 0;
  name: string = '';
  rowsData: any;
  numberAttributes?: NumberAttribute[];
  categoricalAttributes?: CategoricalAttribute[];
}
