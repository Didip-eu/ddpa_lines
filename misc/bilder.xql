xquery version "3.0";
declare namespace atom = "http://www.w3.org/2005/Atom";
declare namespace cei = "http://www.monasterium.net/NS/cei";
declare namespace xrx = "http://www.monasterium.net/NS/xrx";
declare namespace output = "http://www.w3.org/2010/xslt-xquery-serialization";
declare option output:method "xml";
declare option output:item-separator "&#xa;";
(:let $fond := 'AT-ADG/AGDK':)
let $from := 901
let $len := 100
let $where := "fond"
let $dbres := concat('/db/mom-data/metadata.', $where, '.public/')
return
    <charters
        from="{$from}"
        length="{$len}"
        date="{current-dateTime()}"
        xmlns:cei="http://www.monasterium.net/NS/cei">
        {
            for $fonds in distinct-values(subsequence(collection($dbres)//atom:id, $from, $len))
            return
                let $parts := tokenize($fonds, '/')
                let $fond := if ($where = "fond") then
                    (concat($parts[3], '/', $parts[4]))
                else
                    ($parts[3])
                    (:    let $archive := $parts[3]:)
                    (:    let $folder := replace($fond,'[/ ]','_'):)
                let $imagebase := if ($where = "fond")
                then
                    (
                    doc(concat('/db/mom-data/metadata.fond.public/', $fond, '/', $parts[4], '.preferences.xml'))//xrx:param[@name = "image-server-base-url"]/string()
                    )
                else
                    (
                    doc(concat($dbres, $fond, '/', $parts[3], '.cei.xml'))//cei:front/concat('https://', cei:image_server_address, '/', cei:image_server_folder)
                    )
                let $path := concat('/mom-data/metadata.charter.public/', $fond, '/')
                return
                    for $u in collection($path)//atom:entry[.//cei:witnessOrig//cei:graphic/@url != '']
                    let $atomid := $u/atom:id
                    let $issued := $u//cei:issued
                    return
                        <charter
                            id="{$atomid}">
                            {$issued}
                            {
                                for $i in $u//cei:graphic/@url/string()
                                return
                                    <img
                                        src="{concat($imagebase, '/', $i)}"/>
                            }
                        </charter>
                        (:then ((concat( ' &#xA; ', 'mkdir ', $folder),
            for $i in collection(concat('/mom-data/metadata.charter.public/', $fond, '/'))//cei:graphic/@url/string()
            return concat('curl "', $imagebase, '/', $i, '" -o"', $folder,'\',$i, '"'))
        )
        else ():)
        }
    </charters>
