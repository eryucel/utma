import {AttributeCategory} from './attributeCategory';

export class CategoricalAttribute {
  name?: string;
  missing?: number;
  distinct?: number;
  categories?: AttributeCategory[];
}
